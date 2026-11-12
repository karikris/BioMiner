from contextlib import contextmanager
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import polars as pl
import pytest

from biominer.vision.custom_label_run import (
    MediaDownloader, RunState, embedding_key, excel_chunks, export_workbook,
    load_labels, normalize_vectors, run_inference, score_vector, snapshot_inputs,
    text_prototypes,
    input_geolocation_check,
)


@pytest.fixture
def labels():
    return {"version": "test", "templates": ["a photo of {description}", "a closeup of {description}"],
            "heads": {h: {f"label{i}": f"thing{i}" for i in range(3)} for h in ("subject", "context")}}


def test_template_vectors_are_normalized_before_and_after_averaging(labels):
    scorer = SimpleNamespace(embed_text_labels=lambda prompts: [[2, 0, 0], [0, 3, 0]] * 6)
    keys, prototypes, prompts = text_prototypes(scorer, labels)
    assert len(keys) == 6 and len(prompts) == 12
    np.testing.assert_allclose(prototypes[0], [2 ** -0.5, 2 ** -0.5, 0], atol=1e-6)


def test_positive_example_lists_keep_prompt_to_label_alignment(labels):
    labels["heads"]["subject"]["label0"] = ["a beetle", "a bee"]
    prompts_seen = []
    def embed(prompts):
        prompts_seen.extend(prompts)
        return [[1, 0, 0]] * 4 + [[0, 1, 0]] * (len(prompts) - 4)
    keys, prototypes, _ = text_prototypes(SimpleNamespace(embed_text_labels=embed), labels)
    assert keys[0] == ("subject", "label0")
    np.testing.assert_allclose(prototypes[0], [1, 0, 0])
    np.testing.assert_allclose(prototypes[1], [0, 1, 0])
    assert prompts_seen[:4] == ["a photo of a beetle", "a closeup of a beetle", "a photo of a bee", "a closeup of a bee"]


def test_ranks_are_raw_cosines_and_wild_status_remains_unresolved():
    keys = [(h, str(i)) for h in ("subject", "context") for i in range(3)]
    prototypes = normalize_vectors([[1, 0], [0, 1], [-1, 0]] * 2)
    result = score_vector([1, 0], keys, prototypes)
    assert result["bioclip_subject_top1"] == "0"
    assert result["bioclip_subject_margin"] == pytest.approx(1)
    assert result["bioclip_subject_cosine__2"] == pytest.approx(-1)
    assert result["bioclip_wild_status"] == "unresolved"


@pytest.mark.parametrize("vectors", [[[0, 0]], [[float("nan"), 1]]])
def test_invalid_vectors_fail_closed(vectors):
    with pytest.raises(ValueError):
        normalize_vectors(vectors)


def test_label_config_requires_two_independent_heads(tmp_path, labels):
    del labels["heads"]["context"]
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(labels))
    with pytest.raises(ValueError, match="separate subject and context"):
        load_labels(path)


def test_download_falls_back_from_html_and_retains_full_panorama(tmp_path):
    image = Image.new("RGB", (120, 10), "blue")
    image.putpixel((0, 0), (255, 0, 0))
    body = io.BytesIO()
    image.save(body, format="PNG")
    calls = []

    @contextmanager
    def opener(request, timeout):
        calls.append(request.full_url)
        data = b"<html>error</html>" if request.full_url.endswith("large") else body.getvalue()
        response = io.BytesIO(data)
        response.status = 200
        response.headers = {"Content-Type": "text/html" if len(calls) == 1 else "image/png"}
        yield response

    downloader = MediaDownloader(tmp_path, {"001": [("url_l", "https://test/large"), ("url_m", "https://test/medium")]},
                                 opener=opener, attempts=1)
    _, media = downloader.download({"flickr_photo_id": "001"})
    assert media["rendition"] == "url_m"
    with Image.open(media["canonical_path"]) as result:
        assert result.size == (120, 10)
        assert result.getpixel((0, 0)) == (255, 0, 0)
    cached = MediaDownloader(tmp_path, {}, {"001": media}, opener=lambda *a, **k: pytest.fail("cache downloaded again"))
    assert cached.download({"flickr_photo_id": "001"})[1] == media


def test_download_rejects_oversized_body_without_fake_prediction(tmp_path):
    @contextmanager
    def opener(*args, **kwargs):
        response = io.BytesIO(b"x" * 100)
        response.status, response.headers = 200, {"Content-Type": "image/png"}
        yield response
    downloader = MediaDownloader(tmp_path, {}, opener=opener, max_bytes=10, attempts=1)
    _, media = downloader.download({"flickr_photo_id": "001", "image_url": "https://test/image"})
    assert media["status"] == "media_failure"
    assert "byte limit" in media["errors"][0]
    assert "pixel_sha256" not in media


def test_embedding_cache_is_invalidated_by_model_and_preprocessing():
    attestation = {"model_fingerprint": "m1", "preprocessing_fingerprint": "p1"}
    key = embedding_key({"pixel_sha256": "pixels"}, attestation)
    assert key != embedding_key({"pixel_sha256": "pixels"}, dict(attestation, model_fingerprint="m2"))
    assert key != embedding_key({"pixel_sha256": "pixels"}, dict(attestation, preprocessing_fingerprint="p2"))


@pytest.mark.parametrize("row,expected", [
    ({"latitude": 0, "longitude": 0}, "zero_zero_requires_review"),
    ({"latitude": 0, "longitude": 120}, "present_unverified"),
    ({"latitude": -33.8, "longitude": 151.2}, "present_unverified"),
    ({"latitude": 95, "longitude": 120}, "missing_or_invalid_coordinates"),
    ({"latitude": float('nan'), "longitude": 120}, "missing_or_invalid_coordinates"),
    ({"latitude": None, "longitude": None}, "missing_or_invalid_coordinates"),
])
def test_geolocation_checks_flag_null_island_without_rejecting_equator(row, expected):
    assert input_geolocation_check(row) == expected


def test_download_applies_exif_orientation_without_cropping(tmp_path):
    image = Image.new("RGB", (80, 20), "red")
    exif = image.getexif()
    exif[274] = 6  # 90 degrees clockwise.
    body = io.BytesIO()
    image.save(body, format="JPEG", exif=exif)
    @contextmanager
    def opener(*args, **kwargs):
        response = io.BytesIO(body.getvalue())
        response.status, response.headers = 200, {"Content-Type": "image/jpeg"}
        yield response
    downloader = MediaDownloader(tmp_path, {}, opener=opener, attempts=1)
    _, media = downloader.download({"flickr_photo_id": "001", "image_url": "https://test/image"})
    assert (media["width"], media["height"]) == (20, 80)
    with Image.open(media["canonical_path"]) as canonical:
        assert canonical.mode == "RGB"
        assert canonical.width * canonical.height == 1600


def test_memory_pressure_halves_batches_and_keeps_all_photo_bindings(tmp_path, labels):
    class Scorer:
        cache_metrics = {}
        memory_metrics = {}
        def embed_text_labels(self, prompts):
            return [[1, 0, 0]] * len(prompts)
        def embed_image_paths(self, paths):
            if len(paths) > 2:
                raise RuntimeError("MPS out of memory")
            self.last_image_content_hashes = [p.stem for p in paths]
            return [[1, 0, 0]] * len(paths)
    class Downloader:
        def download(self, row):
            pid = row["flickr_photo_id"]
            return pid, {"status": "ready", "pixel_sha256": pid, "canonical_path": pid + ".png"}
    state = RunState(tmp_path / "state.sqlite")
    rows = [{"flickr_photo_id": str(i)} for i in range(4)]
    result = run_inference(rows, state, Downloader(), Scorer(), labels, tmp_path,
                           {"model_fingerprint": "m", "preprocessing_fingerprint": "p"},
                           {"scoring_fingerprint": "oom"}, smoke_size=4)
    assert result["counts"] == {"scored": 4}
    assert result["counters"]["oom_retries"] == 2
    assert result["current_batch_size"] == 2


def test_metadata_join_preserves_review_and_refuses_changed_source(tmp_path):
    source = tmp_path / "source.parquet"
    metadata = tmp_path / "metadata.parquet"
    root = tmp_path / "run"
    root.mkdir()
    pl.DataFrame({"flickr_photo_id": ["001"], "source_record_hash": ["hash"], "raw_title": ["old"]}).write_parquet(source)
    pl.DataFrame({"flickr_photo_id": ["001"], "source_record_hash": ["hash"], "raw_title": ["new"],
                  "has_saved_review": [True]}).write_parquet(metadata)
    rows, _ = snapshot_inputs(source, metadata, root)
    assert rows == [{"flickr_photo_id": "001", "source_record_hash": "hash", "raw_title": "new", "has_saved_review": True}]
    pl.DataFrame({"flickr_photo_id": ["002"]}).write_parquet(source)
    with pytest.raises(ValueError, match="fingerprint changed"):
        snapshot_inputs(source, metadata, root)


def test_excel_preserves_long_unicode_and_does_not_execute_strings(tmp_path):
    value = "🦋" * 17000 + "x" * 33000
    assert "".join(excel_chunks(value)) == value
    assert all(len(v.encode("utf-16-le")) // 2 <= 32767 for v in excel_chunks(value))
    path = tmp_path / "result.xlsx"
    export_workbook(path, [{"flickr_photo_id": "001", "bioclip_status": "media_failure", "comments": value,
                            "title": "=1+1"}], [])
    with zipfile.ZipFile(path) as book:
        xml = ET.fromstring(book.read("xl/worksheets/sheet1.xml"))
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    cells = xml.findall(".//m:row", ns)[1].findall("m:c", ns)
    contents = ["".join(c.itertext()) for c in cells]
    assert "001" in contents and "=1+1" in contents
    assert "".join(c for c in contents if "🦋" in c or c.startswith("x")) == value
    assert xml.find(".//m:f", ns) is None


def test_resume_and_prompt_changes_reuse_embeddings_and_keep_failures(tmp_path, labels):
    class Scorer:
        image_calls = 0
        cache_metrics = {}
        memory_metrics = {}
        def embed_text_labels(self, prompts):
            return [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]] * 2
        def embed_image_paths(self, paths):
            self.image_calls += len(paths)
            self.last_image_content_hashes = [p.stem for p in paths]
            return [[1, 0, 0] for p in paths]
    class Downloader:
        def download(self, row):
            photo_id = row["flickr_photo_id"]
            if photo_id == "003":
                return photo_id, {"status": "media_failure", "errors": ["404"]}
            return photo_id, {"status": "ready", "pixel_sha256": "same-pixels", "canonical_path": "same-pixels.png"}
    rows = [{"flickr_photo_id": str(i).zfill(3), "has_saved_review": True} for i in range(1, 4)]
    state, scorer = RunState(tmp_path / "state.sqlite"), Scorer()
    attestation = {"model_fingerprint": "m", "preprocessing_fingerprint": "p"}
    first = tmp_path / "first"
    first.mkdir()
    manifest = {"scoring_fingerprint": "first"}
    result = run_inference(rows, state, Downloader(), scorer, labels, first, attestation, manifest, smoke_size=3)
    assert scorer.image_calls == 1  # Two photo bindings, one decoded pixel image.
    assert result["counts"] == {"scored": 2, "media_failure": 1}
    assert pl.read_parquet(first / "all_label_scores.parquet").height == 12
    assert pl.read_parquet(first / "image_suggestions.parquet")["has_saved_review"].to_list() == [True] * 3
    run_inference(rows, state, Downloader(), scorer, labels, first, attestation, manifest, smoke_size=3)
    assert scorer.image_calls == 1
    second = tmp_path / "second"
    second.mkdir()
    run_inference(rows, state, Downloader(), scorer, labels, second, attestation,
                  {"scoring_fingerprint": "changed-prompts"}, smoke_size=3)
    assert scorer.image_calls == 1
    assert (second / "image_suggestions.xlsx").is_file()
