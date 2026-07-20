from __future__ import annotations

import sqlite3
from pathlib import Path

from biominer.detection.detector_base import DecodedImage, DetectionCandidate
from biominer.flickr_fetch import yoloe_pilot
from biominer.flickr_fetch.yoloe_pilot import YoloePilotConfig, run_yoloe_pilot


class _FakeDetector:
    model_id = "fake-yoloe"
    model_version = "test"
    checkpoint = "fake.pt"
    prompt_set_fingerprint = "sha256:" + "0" * 64

    def detect_batch(self, images: list[DecodedImage]) -> list[list[DetectionCandidate]]:
        return [[DetectionCandidate("adult_butterfly", 0.9, (0, 0, 1, 1))] for _ in images]

    def close(self) -> None:
        return None


def _source_db(path: Path, count: int) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("""CREATE TABLE source_records (
          source TEXT, flickr_photo_id TEXT, image_url TEXT, image_url_kind TEXT,
          source_record_hash TEXT, query_term TEXT, query_language TEXT, query_field TEXT, created_at TEXT)""")
        conn.executemany("INSERT INTO source_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", [
            ("flickr", str(index), f"https://example.test/{index}.jpg", "large", f"hash-{index}", "butterfly", "en", "text", "2026-07-19")
            for index in range(count)
        ])


def test_pilot_is_deterministic_and_records_visual_screening_only(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.sqlite"
    _source_db(source, 8)
    monkeypatch.setattr(yoloe_pilot, "_download_decode", lambda row, config: DecodedImage(1, 1, "RGB", b"\0\0\0"))
    config = YoloePilotConfig(state_db=source, output_dir=tmp_path / "staging", reports_dir=tmp_path / "reports", sample_size=5)
    result = run_yoloe_pilot(config, detector=_FakeDetector())
    assert result.report["counts"]["classified"] == 5
    assert result.report["butterfly_or_insect_visual_screening_estimate"]["positive_images"] == 5
    assert "not taxonomic validation" in result.report["scientific_scope"]
    first = result.sample_path.read_bytes()
    again = run_yoloe_pilot(config, detector=_FakeDetector())
    assert again.sample_path.read_bytes() == first
