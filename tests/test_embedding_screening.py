from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sqlite3

import polars as pl
import pytest

from biominer.detection.detector_base import DecodedImage
from biominer.detection.embeddinggemma2 import EmbeddingGemma2Config
from biominer.flickr_fetch import embedding_screening as screening


class FakeRouter:
    model_id = "google/embeddinggemma-2"
    model_version = "fixture-runtime"
    fingerprint = "sha256:" + "a" * 64
    prompt_set_fingerprint = "sha256:" + "b" * 64
    calls = 0

    def route_batch(self, images):
        self.calls += 1
        return [{
            "embedding": [1.0] + [0.0] * 767,
            "image_content_hash": "sha256:" + "c" * 64,
            "proposed_domain": "adult_butterfly_field",
            "bioclip_route": "adult_field",
            "routing_action": "review",
            "routing_reason": "provisional_embedding_route",
            "cosine_similarity": 0.7,
            "similarity_margin": 0.2,
            "prompt_similarities_json": "[]",
            "evidence_kind": "full_frame_embedding",
            "localization_status": "unavailable",
            "subject_present": None, "subject_area_ratio": None,
            "detection_count": None, "bbox_xyxy": None,
            "mask_polygon_xyn": None, "detector_score": None,
        } for _ in images]

    def close(self):
        pass


def source_db(path: Path, count=8):
    with sqlite3.connect(path) as conn:
        conn.execute("""CREATE TABLE source_records (
          source TEXT, flickr_photo_id TEXT, image_url TEXT, image_url_kind TEXT,
          source_record_hash TEXT, query_term TEXT, query_language TEXT,
          query_field TEXT, created_at TEXT)""")
        conn.executemany("INSERT INTO source_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", [
            ("flickr", str(i), f"https://example.test/{i}.jpg", "large", f"hash-{i}",
             "butterfly", "en", "text", "2026-11-17") for i in range(count)
        ])


@pytest.fixture
def config(tmp_path, monkeypatch):
    db = tmp_path / "source.sqlite"
    source_db(db)
    monkeypatch.setattr(screening, "_download_decode", lambda row, config:
                        DecodedImage(1, 1, "RGB", b"\0\0\0"))
    return screening.EmbeddingScreeningConfig(
        state_db=db, output_dir=tmp_path / "output", reports_dir=tmp_path / "reports",
        sample_size=5, batch_size=2,
    )


def test_screening_persists_full_frame_evidence_and_resumes_without_inference(config):
    router = FakeRouter()
    result = screening.run_embedding_screening(config, router_factory=lambda _: router)
    rows = pl.read_parquet(result.results_path)
    assert rows.height == 5
    assert rows["detection_count"].null_count() == 5
    assert rows["subject_present"].null_count() == 5
    assert rows["bbox_xyxy"].null_count() == 5
    assert rows["routing_action"].unique().to_list() == ["review"]
    before = router.calls
    screening.run_embedding_screening(config, router_factory=lambda _: router)
    assert router.calls == before


def test_snapshot_resume_rejects_changed_model_policy(config):
    screening.run_embedding_screening(config, router_factory=lambda _: FakeRouter())
    changed = replace(config, model=replace(EmbeddingGemma2Config(), min_margin=0.3))
    with pytest.raises(ValueError, match="manifest|identity"):
        screening.run_embedding_screening(changed, router_factory=lambda _: FakeRouter())


def test_resume_rejects_actual_runtime_change(config):
    screening.run_embedding_screening(config, router_factory=lambda _: FakeRouter())
    changed = FakeRouter()
    changed.fingerprint = "sha256:" + "d" * 64
    with pytest.raises(ValueError, match="runtime|identity"):
        screening.run_embedding_screening(config, router_factory=lambda _: changed)


def test_resume_rejects_unbound_yoloe_state(config):
    config.output_dir.mkdir()
    (config.output_dir / "pilot_state.sqlite").touch()
    with pytest.raises(ValueError, match="unbound|legacy|empty"):
        screening.run_embedding_screening(config, router_factory=lambda _: FakeRouter())


def test_shards_cover_snapshot_once_with_one_router_each(config):
    routers = []

    def factory(_):
        router = FakeRouter()
        routers.append(router)
        return router

    result = screening.run_embedding_screening(replace(config, shard_count=3), router_factory=factory)
    rows = pl.read_parquet(result.results_path)
    assert rows["flickr_photo_id"].n_unique() == 5
    assert rows.group_by("shard_index").len()["len"].sort().to_list() == [1, 2, 2]
    assert len(routers) == 3


def test_decode_failure_is_operational_evidence(config, monkeypatch):
    def fail(*_):
        raise OSError("fixture decode failure")

    monkeypatch.setattr(screening, "_download_decode", fail)
    result = screening.run_embedding_screening(replace(config, max_attempts=1), router_factory=lambda _: FakeRouter())
    assert pl.read_parquet(result.results_path).is_empty()
    assert pl.read_parquet(result.failures_path).height == 5
    assert result.report["status"] == "complete_with_operational_failures"
    assert result.report["counts"]["image_routes"] == 0


def test_snapshot_tamper_with_same_row_count_is_rejected(config):
    result = screening.run_embedding_screening(config, router_factory=lambda _: FakeRouter())
    snapshot = result.sample_path
    pl.read_parquet(snapshot).with_columns(pl.lit("changed").alias("image_url")).write_parquet(snapshot)
    with pytest.raises(ValueError, match="snapshot|checksum"):
        screening.run_embedding_screening(config, router_factory=lambda _: FakeRouter())


def test_failed_batch_never_publishes_success_or_biological_negative(config):
    class FailingRouter(FakeRouter):
        def route_batch(self, images):
            raise RuntimeError("fixture model failure")

    with pytest.raises(RuntimeError, match="restart budget"):
        screening.run_embedding_screening(replace(config, max_attempts=2), router_factory=lambda _: FailingRouter())
    assert not (config.output_dir / "embedding_screening_manifest.json").exists()
    with sqlite3.connect(config.output_dir / "shard_000/embedding_state.sqlite") as conn:
        assert conn.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 0


def test_model_failure_restarts_router_before_routing_later_images(config):
    routers = []

    class BrokenRouter(FakeRouter):
        def route_batch(self, images):
            raise RuntimeError("worker closed after timeout")

    def factory(_):
        router = BrokenRouter() if not routers else FakeRouter()
        routers.append(router)
        return router

    result = screening.run_embedding_screening(config, router_factory=factory)
    assert len(routers) == 2
    assert pl.read_parquet(result.results_path).height == 5
    assert pl.read_parquet(result.failures_path).is_empty()


def test_manifest_pins_exact_revision(config):
    result = screening.run_embedding_screening(config, router_factory=lambda _: FakeRouter())
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["identity"]["model"]["revision"] == EmbeddingGemma2Config().revision
    assert len(manifest["identity"]["model"]["revision"]) == 40


def test_screening_rejects_concurrent_writer_to_same_output(config):
    def factory(_):
        with pytest.raises(RuntimeError, match="active|locked"):
            screening.run_embedding_screening(config, router_factory=lambda _: FakeRouter())
        return FakeRouter()

    screening.run_embedding_screening(config, router_factory=factory)
