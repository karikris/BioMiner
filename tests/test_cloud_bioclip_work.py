from __future__ import annotations

from pathlib import Path

import polars as pl

from biominer.bioclip.candidate_sets import build_candidate_set
from biominer.bioclip.cloud_work import bioclip_score_work_item, enqueue_bioclip_work_from_detection_shards, run_cloud_bioclip_batch
from biominer.run.stages import RunStage
from biominer.species.context import CommonName, SpeciesContext
from biominer.workstore.sqlite import SQLiteWorkStore


def test_enqueue_bioclip_work_from_detection_shards_only_uses_detected_butterflies(tmp_path: Path) -> None:
    storage = _FakeCloudStorage()
    workstore = SQLiteWorkStore(tmp_path / "workstore.sqlite")
    detection_uri = "s3://biominer/runs/run_id=run-1/staging/evidence/stage=detect_objects/run_id=run-1/worker=w1/batch=001.parquet"
    storage.parquet_payloads[detection_uri] = pl.DataFrame(
        [
            _detection_row("photo-1", "det-1", "sha256:crop-1", "butterfly_like", "detected"),
            _detection_row("photo-2", "det-2", "sha256:crop-2", "hard_negative", "detected"),
            _detection_row("photo-3", "det-3", "sha256:crop-3", "moth_like", "detected"),
            _detection_row("photo-4", "det-4", "", "butterfly_like", "no_detection"),
        ]
    )
    workstore.register_shard(
        job_name="biominer_production_run",
        registry_version="registry-v1",
        stage=RunStage.DETECT_OBJECTS.value,
        run_id="run-1",
        worker_id="detector-1",
        uri=detection_uri,
        checksum=None,
        row_count=4,
    )

    first = enqueue_bioclip_work_from_detection_shards(
        storage=storage,
        workstore=workstore,
        job_name="biominer_production_run",
        registry_version="registry-v1",
        run_id="run-1",
        detection_stage=RunStage.DETECT_OBJECTS.value,
        score_stage=RunStage.SCORE_BIOCLIP.value,
        model_id="imageomics/bioclip-2.5-vith14",
        model_version="2.5",
        model_checkpoint="bioclip-2.5",
        candidate_set_id="candidate-set-1",
        ablation_modes=("detector_crop",),
    )
    second = enqueue_bioclip_work_from_detection_shards(
        storage=storage,
        workstore=workstore,
        job_name="biominer_production_run",
        registry_version="registry-v1",
        run_id="run-1",
        detection_stage=RunStage.DETECT_OBJECTS.value,
        score_stage=RunStage.SCORE_BIOCLIP.value,
        model_id="imageomics/bioclip-2.5-vith14",
        model_version="2.5",
        model_checkpoint="bioclip-2.5",
        candidate_set_id="candidate-set-1",
        ablation_modes=("detector_crop",),
    )

    assert first.detection_shards_seen == 1
    assert first.detections_seen == 4
    assert first.eligible_detections_seen == 1
    assert first.enqueued_work_items == 1
    assert first.duplicate_work_items == 0
    assert second.enqueued_work_items == 0
    assert second.duplicate_work_items == 1
    items = workstore.list_work_items(
        job_name="biominer_production_run",
        stage=RunStage.SCORE_BIOCLIP.value,
        registry_version="registry-v1",
    )
    assert [item["status"] for item in items] == ["pending"]
    payload = items[0]["payload"]
    assert payload["detection_shard_uri"] == detection_uri
    assert payload["ablation_mode"] == "detector_crop"
    assert payload["candidate_set_id"] == "candidate-set-1"
    assert payload["model"]["checkpoint"] == "bioclip-2.5"
    assert payload["detection"]["flickr_photo_id"] == "photo-1"
    assert payload["detection"]["detector_label"] == "butterfly_like"


def test_run_cloud_bioclip_batch_chunks_detector_crops_by_crop_batch_size() -> None:
    class BatchRecordingScorer:
        model_id = "fake-bioclip"
        model_version = "test"
        model_checkpoint = "fake-checkpoint"

        def __init__(self) -> None:
            self.initial_batches: list[tuple[str, ...]] = []

        def score(self, item, labels):  # noqa: ANN001, ANN202 - proves the batch path is used.
            raise AssertionError(f"unexpected single-item BioCLIP score for {item.get('detection_id')}")

        def score_label_sets_batch(self, items, label_sets):  # noqa: ANN001, ANN202 - mirrors object batch scorer API.
            if "species" in label_sets:
                self.initial_batches.append(tuple(str(item["detection_id"]) for item in items))
            return {
                name: [
                    {label: (0.83 if label == "a photo of Danaus plexippus" else 0.1) for label in labels}
                    for _item in items
                ]
                for name, labels in label_sets.items()
            }

    context = _context()
    candidate_set = build_candidate_set(context, allow_single_target_fixture=True)
    scorer = BatchRecordingScorer()
    work_items = []
    for index in range(5):
        payload = bioclip_score_work_item(
            _detection_row(f"photo-{index}", f"det-{index}", f"sha256:crop-{index}", "butterfly_like", "detected"),
            run_id="run-1",
            detection_shard_uri="s3://biominer/detections.parquet",
            model={"model_id": "fake-bioclip", "model_version": "test", "checkpoint": "fake-checkpoint"},
            candidate_set_id=candidate_set.candidate_set_id,
            ablation_mode="detector_crop",
        )
        work_items.append({"work_key": payload["work_key"], "payload": payload})

    result = run_cloud_bioclip_batch(
        work_items=work_items,
        species_context=context,
        candidate_set=candidate_set,
        scorer=scorer,
        crop_batch_size=2,
    )

    assert scorer.initial_batches == [("det-0", "det-1"), ("det-2", "det-3"), ("det-4",)]
    assert result.work_items_seen == 5
    assert result.detections_seen == 5
    assert result.crops_scored == 5


def _context() -> SpeciesContext:
    return SpeciesContext(
        scientific_name="Danaus plexippus",
        accepted_taxon_key="gbif:5131654",
        canonical_name="Danaus plexippus",
        family="Nymphalidae",
        genus="Danaus",
        family_key="gbif:7017",
        genus_key="gbif:1927164",
        species_key="gbif:5131654",
        registry_version="registry-v1",
        synonyms=("Anosia plexippus",),
        common_names=(CommonName(name="monarch butterfly", language="en", source="gbif"),),
        regions=(),
    )


def _detection_row(photo_id: str, detection_id: str, crop_hash: str, label: str, status: str) -> dict[str, object]:
    return {
        "source": "flickr",
        "flickr_photo_id": photo_id,
        "source_record_hash": f"sha256:source-{photo_id}",
        "image_url": f"https://live.staticflickr.com/{photo_id}.jpg",
        "photo_page_url": f"https://www.flickr.com/photos/u/{photo_id}",
        "detection_id": detection_id,
        "detector_backend": "fake",
        "prediction_source": "object_detector:fake",
        "detector_model_id": "fake-detector",
        "detector_model_version": "test",
        "detector_checkpoint": "fake-checkpoint",
        "detected_at": "2026-01-01T00:00:00+00:00",
        "bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
        "bbox_xyxyn": [0.0, 0.0, 1.0, 1.0],
        "bbox_xywhn": [0.5, 0.5, 1.0, 1.0],
        "box_area_ratio": 0.5,
        "detector_label": label,
        "detector_score": 0.91,
        "objectness_score": 0.91,
        "nms_group_id": None,
        "crop_padding_ratio": 0.12,
        "crop_hash": crop_hash,
        "crop_width": 336,
        "crop_height": 336,
        "crop_storage_policy": "ephemeral",
        "detection_status": status,
        "failure_reason": None if status == "detected" else "no_butterfly_like_object",
        "schema_version": "object-detection-v1",
    }


class _FakeCloudStorage:
    def __init__(self) -> None:
        self.parquet_payloads: dict[str, pl.DataFrame] = {}

    def read_parquet(self, uri: str) -> pl.DataFrame:
        return self.parquet_payloads[uri]
