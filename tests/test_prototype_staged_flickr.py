from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from PIL import Image
import polars as pl

from biominer.benchmarks.prototype_staged_flickr import (
    SCORE_SEMANTICS,
    PrototypeStagedFlickrConfig,
    StagedFlickrImage,
    run_prototype_staged_flickr,
)
from biominer.bioclip.bioclip_worker import decoded_rgb_image_content_hash
from biominer.detection.detector_base import DecodedImage, DetectionCandidate


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


class FakeFetcher:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def fetch(self, photo_id: str, cache_dir: Path) -> StagedFlickrImage:
        self.calls.append(photo_id)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{photo_id}.png"
        value = int(photo_id) % 255
        Image.new("RGB", (4, 3), (value, 20, 30)).save(path)
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            decoded_hash = decoded_rgb_image_content_hash(rgb)
            decoded = DecodedImage(4, 3, "RGB", rgb.tobytes(), source_uri="fixture")
        return StagedFlickrImage(
            flickr_photo_id=photo_id,
            image_url=f"https://example.invalid/{photo_id}.png",
            path=path,
            source_image_sha256=_sha256(path),
            decoded_image_sha256=decoded_hash,
            width=4,
            height=3,
            decoded=decoded,
        )


class FailOnceFetcher(FakeFetcher):
    def __init__(self, failed_photo_id: str) -> None:
        super().__init__()
        self.failed_photo_id = failed_photo_id
        self.failed = False

    def fetch(self, photo_id: str, cache_dir: Path) -> StagedFlickrImage:
        if photo_id == self.failed_photo_id and not self.failed:
            self.failed = True
            self.calls.append(photo_id)
            raise OSError("temporary source failure")
        return super().fetch(photo_id, cache_dir)


class AlwaysFailFetcher(FakeFetcher):
    def __init__(self, failed_photo_id: str) -> None:
        super().__init__()
        self.failed_photo_id = failed_photo_id

    def fetch(self, photo_id: str, cache_dir: Path) -> StagedFlickrImage:
        if photo_id == self.failed_photo_id:
            self.calls.append(photo_id)
            raise OSError("persistent source failure")
        return super().fetch(photo_id, cache_dir)


class FakeScorer:
    def __init__(self) -> None:
        self.last_image_content_hashes: list[str] | None = None
        self.worker_process_starts = 0
        self.model_load_count = 0
        self.model_cache_hit_count = 0
        self.model_refresh_count = 0
        self.device = "mps"
        self.gpu_name = "Fake MPS"
        self.effective_image_resize_mode = "longest"
        self.model_weights_sha256 = "sha256:" + "1" * 64
        self.open_clip_config_sha256 = "sha256:" + "2" * 64
        self.preprocessing_version = "test-preprocess-v1"
        self.preprocessing_fingerprint = "sha256:" + "3" * 64
        self.image_calls = 0
        self.text_calls = 0

    @property
    def cache_metrics(self) -> dict[str, int | float | bool]:
        return {
            "bioclip_worker_process_starts": self.worker_process_starts,
            "bioclip_worker_requests": self.text_calls + self.image_calls + 1,
            "bioclip_model_loads": self.model_load_count,
            "bioclip_model_cache_hits": self.model_cache_hit_count,
            "bioclip_model_refreshes": self.model_refresh_count,
            "bioclip_model_cache_hit_rate": 0.5,
            "bioclip_last_request_cache_hit": True,
        }

    @property
    def memory_metrics(self) -> dict[str, int]:
        return {
            "mps_current_allocated_memory": 1024,
            "mps_driver_allocated_memory": 2048,
            "mps_recommended_max_memory": 4096,
            "mps_peak_current_allocated_memory": 1536,
            "mps_peak_driver_allocated_memory": 2560,
        }

    def pin_reference_model_identity(self, **kwargs: str) -> None:
        assert kwargs["image_resize_mode"] == "longest"

    def ensure_model_attestation(self) -> None:
        self.worker_process_starts = 1
        self.model_load_count = 1

    def embed_text_labels(self, labels: list[str]) -> list[list[float]]:
        self.text_calls += 1
        self.model_cache_hit_count += 1
        output = []
        for label in labels:
            if "Papilio demoleus" in label:
                output.append([1.0, 0.0])
            elif "Papilio polytes" in label:
                output.append([0.0, 1.0])
            else:
                output.append([0.6, 0.8])
        return output

    def embed_image_paths(self, image_paths: list[Path]) -> list[list[float]]:
        self.image_calls += 1
        self.model_cache_hit_count += 1
        hashes = []
        for path in image_paths:
            with Image.open(path) as image:
                hashes.append(decoded_rgb_image_content_hash(image.convert("RGB")))
        self.last_image_content_hashes = hashes
        return [[1.0, 0.0] for _ in image_paths]

    def close(self) -> None:
        return None


class FakeDetector:
    def __init__(self) -> None:
        self.worker_process_starts = 0
        self.worker_request_count = 0
        self.model_id = "fake-yoloe"
        self.model_version = "test"
        self.checkpoint = "yoloe-26s-seg.pt"
        self.prompt_set_fingerprint = "sha256:" + "4" * 64

    def detect_batch(
        self, images: list[DecodedImage]
    ) -> list[list[DetectionCandidate]]:
        self.worker_process_starts = 1
        self.worker_request_count += 1
        return [
            [
                DetectionCandidate(
                    "butterfly_like",
                    0.9,
                    (0.0, 0.0, 4.0, 3.0),
                    detector_prompt="adult butterfly",
                )
            ]
            for _ in images
        ]

    def close(self) -> None:
        return None


def _write_inputs(tmp_path: Path) -> PrototypeStagedFlickrConfig:
    ids = ["10", "20", "30", "40"]
    geography = tmp_path / "geography.parquet"
    pl.DataFrame(
        {
            "flickr_photo_id": ids,
            "source_record_hash": [f"sha256:{value:0>64}" for value in ids],
            "coordinate_quality": ["flickr_city"] * 4,
        }
    ).write_parquet(geography)
    assignments = tmp_path / "assignments.parquet"
    pl.DataFrame(
        {"flickr_photo_id": ids, "geo_cluster_id": ["geo:test"] * 4}
    ).write_parquet(assignments)
    query_hits = tmp_path / "hits.parquet"
    pl.DataFrame(
        {
            "flickr_photo_id": ids,
            "search_term": ["lime butterfly"] * 4,
            "query_tier": ["common_name:high:text"] * 4,
        }
    ).write_parquet(query_hits)
    competitors = tmp_path / "competitors.parquet"
    pl.DataFrame(
        {
            "candidate_rank": [1],
            "candidate_accepted_taxon_key": ["gbif:1938088"],
            "candidate_scientific_name": ["Papilio polytes"],
            "candidate_reason": ["regional_same_genus_occurrence_overlap"],
        }
    ).write_parquet(competitors)
    readiness = tmp_path / "readiness.json"
    readiness.write_text(
        json.dumps(
            {
                "bank_status": "prototype_only",
                "classification_authorised": True,
            }
        ),
        encoding="utf-8",
    )
    embeddings = tmp_path / "embeddings.parquet"
    pl.DataFrame(
        {
            "reference_media_id": ["target-ref", "competitor-ref"],
            "accepted_taxon_key": ["gbif:1938069", "gbif:1938088"],
            "scientific_name": ["Papilio demoleus", "Papilio polytes"],
            "reference_group": ["target_adult", "selected_regional_competitors"],
            "route": ["adult_field", "adult_field"],
            "dataset_split": ["support_train", "support_train"],
            "embedding": [[1.0, 0.0], [0.0, 1.0]],
            "model_id": ["imageomics/bioclip-2.5-vith14"] * 2,
            "model_revision": ["a" * 40] * 2,
            "model_weights_sha256": ["sha256:" + "1" * 64] * 2,
            "open_clip_version": ["3.3.0"] * 2,
            "open_clip_config_sha256": ["sha256:" + "2" * 64] * 2,
            "preprocessing_fingerprint": ["sha256:" + "3" * 64] * 2,
        }
    ).write_parquet(embeddings)
    prototypes = tmp_path / "prototypes.parquet"
    pl.DataFrame(
        {
            "accepted_taxon_key": ["gbif:1938069", "gbif:1938088"],
            "route": ["adult_field", "adult_field"],
            "scope_type": ["global", "global"],
            "geo_cluster_id": ["all", "all"],
            "embedding": [[1.0, 0.0], [0.0, 1.0]],
        }
    ).write_parquet(prototypes)
    return PrototypeStagedFlickrConfig(
        geography=geography,
        geography_sha256=_sha256(geography),
        assignments=assignments,
        assignments_sha256=_sha256(assignments),
        query_hits=query_hits,
        query_hits_sha256=_sha256(query_hits),
        regional_competitors=competitors,
        regional_competitors_sha256=_sha256(competitors),
        readiness=readiness,
        readiness_sha256=_sha256(readiness),
        reference_embeddings=embeddings,
        reference_embeddings_sha256=_sha256(embeddings),
        reference_prototypes=prototypes,
        reference_prototypes_sha256=_sha256(prototypes),
        output_dir=tmp_path / "output",
        bioclip_runtime_python=tmp_path / "bioclip-python",
        bioclip_hf_cache_dir=tmp_path / "hf-cache",
        yoloe_runtime_python=tmp_path / "yolo-python",
        model_name="imageomics/bioclip-2.5-vith14",
        model_revision="a" * 40,
        open_clip_version="3.3.0",
        stage_limits=(2, 4),
        target_accepted_taxon_key="gbif:1938069",
        target_scientific_name="Papilio demoleus",
        bioclip_batch_size=2,
        yoloe_batch_size=2,
    )


def test_staged_flickr_runner_scores_complete_union_and_resumes(tmp_path: Path) -> None:
    config = _write_inputs(tmp_path)
    fetcher = FakeFetcher()
    scorer = FakeScorer()
    detector = FakeDetector()

    result = run_prototype_staged_flickr(
        config, flickr_fetcher=fetcher, scorer=scorer, detector=detector
    )

    assert result.report["status"] == "passed"
    assert [stage["cumulative_limit"] for stage in result.report["stages"]] == [2, 4]
    assert result.report["storage"] == {
        "backend": "local",
        "s3_permitted": False,
        "s3_accessed": False,
    }
    assert result.report["memory"]["mps_recommended_max_memory"] == 4096
    rows = pl.read_parquet(result.results_path)
    assert rows.height == 4
    assert rows["target_scored"].all()
    assert not rows["higher_rank_pruning_applied"].any()
    assert not rows["spatial_crop_applied"].any()
    assert set(rows["reference_route_used"]) == {"adult_field"}
    assert set(rows["score_semantics"]) == {SCORE_SEMANTICS}
    candidates = pl.read_parquet(result.candidates_path)
    expected_species = rows["regional_scored_count"][0]
    assert (
        candidates.filter(pl.col("class_kind") == "species").height
        == rows.height * expected_species
    )
    assert not (config.output_dir / "cache" / "images").exists()

    no_fetch = FakeFetcher()
    no_model = FakeScorer()
    no_detector = FakeDetector()
    resumed = run_prototype_staged_flickr(
        config,
        flickr_fetcher=no_fetch,
        scorer=no_model,
        detector=no_detector,
    )
    assert resumed.report["report_fingerprint"] == result.report["report_fingerprint"]
    assert no_fetch.calls == []
    assert no_model.text_calls == 0
    assert no_model.image_calls == 0
    assert no_detector.worker_request_count == 0


def test_staged_flickr_config_rejects_s3() -> None:
    values = {
        "geography": Path("g"),
        "geography_sha256": "sha256:" + "0" * 64,
        "assignments": Path("a"),
        "assignments_sha256": "sha256:" + "0" * 64,
        "query_hits": Path("q"),
        "query_hits_sha256": "sha256:" + "0" * 64,
        "regional_competitors": Path("c"),
        "regional_competitors_sha256": "sha256:" + "0" * 64,
        "readiness": Path("r"),
        "readiness_sha256": "sha256:" + "0" * 64,
        "reference_embeddings": Path("e"),
        "reference_embeddings_sha256": "sha256:" + "0" * 64,
        "reference_prototypes": Path("p"),
        "reference_prototypes_sha256": "sha256:" + "0" * 64,
        "output_dir": Path("o"),
        "bioclip_runtime_python": Path("bp"),
        "bioclip_hf_cache_dir": Path("hc"),
        "yoloe_runtime_python": Path("yp"),
        "model_name": "model",
        "model_revision": "a" * 40,
        "open_clip_version": "3.3.0",
        "stage_limits": (100, 1000),
        "target_accepted_taxon_key": "gbif:1938069",
        "target_scientific_name": "Papilio demoleus",
        "storage_backend": "s3",
        "s3_permitted": True,
    }
    try:
        PrototypeStagedFlickrConfig(**values)
    except ValueError as exc:
        assert "local-only storage" in str(exc)
    else:
        raise AssertionError("S3 config must be rejected")


def test_staged_flickr_retry_reuses_prior_completed_gate(tmp_path: Path) -> None:
    config = replace(_write_inputs(tmp_path), max_failure_rate=0.5)
    fetcher = FailOnceFetcher("40")
    first = run_prototype_staged_flickr(
        config,
        flickr_fetcher=fetcher,
        scorer=FakeScorer(),
        detector=FakeDetector(),
    )
    assert first.report["counts"]["failures"] == 1

    second = run_prototype_staged_flickr(
        config,
        flickr_fetcher=fetcher,
        scorer=FakeScorer(),
        detector=FakeDetector(),
    )
    assert second.report["counts"]["classified"] == 4
    assert second.report["counts"]["failures"] == 0
    assert second.report["stages"][0]["resumed_without_stage_work"] is True
    assert "resumed_records_classified" not in second.report["stages"][0]
    assert "resume_validation_checks" not in second.report["stages"][0]
    assert second.report["stages"][1]["status"] == "passed"
    assert second.report["stages"][1]["resumed_records_classified"] == 1
    assert "resumed_without_stage_work" not in second.report["stages"][1]
    assert "resumed_retry_without_new_classification" not in second.report["stages"][1]


def test_staged_flickr_retry_with_no_new_image_reuses_gate_evidence(
    tmp_path: Path,
) -> None:
    config = replace(_write_inputs(tmp_path), max_failure_rate=0.5)
    fetcher = AlwaysFailFetcher("40")
    first = run_prototype_staged_flickr(
        config,
        flickr_fetcher=fetcher,
        scorer=FakeScorer(),
        detector=FakeDetector(),
    )
    assert first.report["counts"]["failures"] == 1

    second_detector = FakeDetector()
    second = run_prototype_staged_flickr(
        config,
        flickr_fetcher=fetcher,
        scorer=FakeScorer(),
        detector=second_detector,
    )
    assert second.report["counts"]["failures"] == 1
    assert (
        second.report["stages"][1]["resumed_retry_without_new_classification"] is True
    )
    assert second_detector.worker_request_count == 0
