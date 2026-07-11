from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

import biominer.bioclip.cloud_work as cloud_work
from biominer.bioclip.candidate_sets import build_candidate_set
from biominer.bioclip.classification_modes import HIERARCHICAL_BUTTERFLY_CLASSIFICATION
from biominer.bioclip.cloud_work import bioclip_score_work_item, enqueue_bioclip_work_from_detection_shards, run_cloud_bioclip_batch
from biominer.benchmarks.vision_plumbing import benchmark_taxonomy_store
from biominer.registry.classification_v2 import (
    CLASSIFICATION_V2_PROMPT_VERSION as PROMPT_VARIANT_VERSION,
    CLASSIFICATION_V2_VERSION as CLASSIFICATION_TABLE_VERSION,
)
from biominer.registry.classification_v3 import (
    CLASSIFICATION_V3_PROMPT_VERSION,
    CLASSIFICATION_V3_VERSION,
)
from biominer.run.stages import RunStage
from biominer.species.context import CommonName, SpeciesContext
from biominer.vision.gates import BioClipGateMode, BioClipGatePolicy
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
    assert payload["classification_mode"] == "target_scope_object_screening"
    assert payload["model"]["checkpoint"] == "bioclip-2.5"
    assert payload["detection"]["flickr_photo_id"] == "photo-1"
    assert payload["detection"]["detector_label"] == "butterfly_like"


def test_enqueue_bioclip_work_can_use_exclude_hard_negative_gate(tmp_path: Path) -> None:
    storage = _FakeCloudStorage()
    workstore = SQLiteWorkStore(tmp_path / "workstore.sqlite")
    detection_uri = "s3://biominer/runs/run_id=run-1/staging/evidence/stage=detect_objects/run_id=run-1/worker=w1/batch=001.parquet"
    storage.parquet_payloads[detection_uri] = pl.DataFrame(
        [
            _detection_row("photo-butterfly", "det-butterfly", "sha256:crop-butterfly", "butterfly_like", "detected"),
            _detection_row("photo-moth", "det-moth", "sha256:crop-moth", "moth_like", "detected"),
            _detection_row("photo-caterpillar", "det-caterpillar", "sha256:crop-caterpillar", "caterpillar", "detected"),
            _detection_row("photo-pupa", "det-pupa", "sha256:crop-pupa", "pupa", "detected"),
            _detection_row("photo-insect", "det-insect", "sha256:crop-insect", "insect_like", "detected"),
            _detection_row("photo-hard-negative", "det-hard-negative", "sha256:crop-hard-negative", "hard_negative", "detected"),
            _detection_row("photo-empty", "det-empty", "", "no_detection", "no_detection"),
            _detection_row("photo-failed", "det-failed", "", "failed_image_load", "failed_image_load"),
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
        row_count=8,
    )

    result = enqueue_bioclip_work_from_detection_shards(
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
        bioclip_gate_policy=BioClipGatePolicy(
            mode=BioClipGateMode.EXCLUDE_HARD_NEGATIVE,
            score_no_detection_whole_image=True,
        ),
    )

    assert result.detections_seen == 8
    assert result.eligible_detections_seen == 6
    assert result.enqueued_work_items == 6
    items = workstore.list_work_items(
        job_name="biominer_production_run",
        stage=RunStage.SCORE_BIOCLIP.value,
        registry_version="registry-v1",
    )
    payloads = [item["payload"] for item in items]
    assert [payload["detection_id"] for payload in payloads] == [
        "det-butterfly",
        "det-moth",
        "det-caterpillar",
        "det-pupa",
        "det-insect",
        "det-empty",
    ]
    assert [payload["ablation_mode"] for payload in payloads] == [
        "detector_crop",
        "detector_crop",
        "detector_crop",
        "detector_crop",
        "detector_crop",
        "whole_image",
    ]
    assert {payload["bioclip_gate_mode"] for payload in payloads} == {"exclude_hard_negative"}
    assert payloads[-1]["bioclip_gate_reason"] == "no_detection_whole_image_fallback"


def test_bioclip_work_item_key_changes_by_classification_mode_and_taxonomy_version() -> None:
    detection = _detection_row("photo-1", "det-1", "sha256:crop-1", "butterfly_like", "detected")
    target = bioclip_score_work_item(
        detection,
        run_id="run-1",
        detection_shard_uri="s3://biominer/detections.parquet",
        model={"model_id": "fake-bioclip", "model_version": "test", "checkpoint": "fake-checkpoint"},
        candidate_set_id="candidate-set-1",
        ablation_mode="detector_crop",
    )
    hierarchical = bioclip_score_work_item(
        detection,
        run_id="run-1",
        detection_shard_uri="s3://biominer/detections.parquet",
        model={"model_id": "fake-bioclip", "model_version": "test", "checkpoint": "fake-checkpoint"},
        candidate_set_id="candidate-set-1",
        ablation_mode="detector_crop",
        classification_mode=HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        taxonomy_table_version=CLASSIFICATION_TABLE_VERSION,
        taxonomy_prompt_variant_version=PROMPT_VARIANT_VERSION,
    )

    assert target["work_key"] != hierarchical["work_key"]
    assert hierarchical["classification_mode"] == HIERARCHICAL_BUTTERFLY_CLASSIFICATION
    assert hierarchical["taxonomy_table_version"] == CLASSIFICATION_TABLE_VERSION
    assert hierarchical["taxonomy_prompt_variant_version"] == PROMPT_VARIANT_VERSION


def test_bioclip_work_item_key_changes_by_model_top_k_and_crop_identity() -> None:
    detection = _detection_row("photo-1", "det-1", "sha256:crop-1", "butterfly_like", "detected")
    base = bioclip_score_work_item(
        detection,
        run_id="run-1",
        detection_shard_uri="s3://biominer/detections.parquet",
        model={"model_id": "fake-bioclip", "model_version": "test", "checkpoint": "fake-checkpoint"},
        candidate_set_id="candidate-set-1",
        ablation_mode="detector_crop",
        classification_mode=HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        taxonomy_table_version=CLASSIFICATION_TABLE_VERSION,
        taxonomy_prompt_variant_version=PROMPT_VARIANT_VERSION,
        family_top_k=3,
        species_first_pass_top_k=20,
        species_rerank_top_k=20,
    )
    model_changed = bioclip_score_work_item(
        detection,
        run_id="run-1",
        detection_shard_uri="s3://biominer/detections.parquet",
        model={"model_id": "fake-bioclip-large", "model_version": "test", "checkpoint": "fake-checkpoint"},
        candidate_set_id="candidate-set-1",
        ablation_mode="detector_crop",
        classification_mode=HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        taxonomy_table_version=CLASSIFICATION_TABLE_VERSION,
        taxonomy_prompt_variant_version=PROMPT_VARIANT_VERSION,
        family_top_k=3,
        species_first_pass_top_k=20,
        species_rerank_top_k=20,
    )
    taxonomy_changed = bioclip_score_work_item(
        detection,
        run_id="run-1",
        detection_shard_uri="s3://biominer/detections.parquet",
        model={"model_id": "fake-bioclip", "model_version": "test", "checkpoint": "fake-checkpoint"},
        candidate_set_id="candidate-set-1",
        ablation_mode="detector_crop",
        classification_mode=HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        taxonomy_table_version="classification-table-v-next",
        taxonomy_prompt_variant_version=PROMPT_VARIANT_VERSION,
        family_top_k=3,
        species_first_pass_top_k=20,
        species_rerank_top_k=20,
    )
    top_k_changed = bioclip_score_work_item(
        detection,
        run_id="run-1",
        detection_shard_uri="s3://biominer/detections.parquet",
        model={"model_id": "fake-bioclip", "model_version": "test", "checkpoint": "fake-checkpoint"},
        candidate_set_id="candidate-set-1",
        ablation_mode="detector_crop",
        classification_mode=HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        taxonomy_table_version=CLASSIFICATION_TABLE_VERSION,
        taxonomy_prompt_variant_version=PROMPT_VARIANT_VERSION,
        family_top_k=5,
        species_first_pass_top_k=20,
        species_rerank_top_k=20,
    )
    crop_changed = bioclip_score_work_item(
        {**detection, "crop_hash": "sha256:crop-2", "crop_padding_ratio": 0.18},
        run_id="run-1",
        detection_shard_uri="s3://biominer/detections.parquet",
        model={"model_id": "fake-bioclip", "model_version": "test", "checkpoint": "fake-checkpoint"},
        candidate_set_id="candidate-set-1",
        ablation_mode="detector_crop",
        classification_mode=HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        taxonomy_table_version=CLASSIFICATION_TABLE_VERSION,
        taxonomy_prompt_variant_version=PROMPT_VARIANT_VERSION,
        family_top_k=3,
        species_first_pass_top_k=20,
        species_rerank_top_k=20,
    )

    assert base["work_key"] != model_changed["work_key"]
    assert base["work_key"] != taxonomy_changed["work_key"]
    assert base["work_key"] != top_k_changed["work_key"]
    assert base["work_key"] != crop_changed["work_key"]
    assert base["top_k_settings"] == {
        "family_top_k": 3,
        "species_first_pass_top_k": 20,
        "species_rerank_top_k": 20,
    }


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


def test_run_cloud_bioclip_batch_adaptive_batching_halves_after_memory_error() -> None:
    class AdaptiveScorer:
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
                if len(items) > 12:
                    raise RuntimeError("CUDA out of memory while scoring BioCLIP crop batch")
            return {
                name: [
                    {label: (0.83 if label == "a photo of Danaus plexippus" else 0.1) for label in labels}
                    for _item in items
                ]
                for name, labels in label_sets.items()
            }

    context = _context()
    candidate_set = build_candidate_set(context, allow_single_target_fixture=True)
    work_items = [
        {
            "work_key": payload["work_key"],
            "payload": payload,
        }
        for index in range(24)
        for payload in [
            bioclip_score_work_item(
                _detection_row(f"photo-{index}", f"det-{index}", f"sha256:crop-{index}", "butterfly_like", "detected"),
                run_id="run-1",
                detection_shard_uri="s3://biominer/detections.parquet",
                model={"model_id": "fake-bioclip", "model_version": "test", "checkpoint": "fake-checkpoint"},
                candidate_set_id=candidate_set.candidate_set_id,
                ablation_mode="detector_crop",
            )
        ]
    ]
    scorer = AdaptiveScorer()

    result = run_cloud_bioclip_batch(
        work_items=work_items,
        species_context=context,
        candidate_set=candidate_set,
        scorer=scorer,
        crop_batch_size=24,
        adaptive_batching=True,
        min_crop_batch_size=1,
    )

    assert result.crops_scored == 24
    assert result.adaptive_batching_enabled is True
    assert result.bioclip_batch_retries == 1
    assert result.bioclip_batch_size_initial == 24
    assert result.bioclip_batch_size_final == 12
    assert result.bioclip_batch_size_min == 1
    assert [len(batch) for batch in scorer.initial_batches] == [24, 12, 12]


def test_run_cloud_bioclip_batch_adaptive_batching_does_not_retry_non_memory_error() -> None:
    class NonMemoryScorer:
        model_id = "fake-bioclip"
        model_version = "test"
        model_checkpoint = "fake-checkpoint"

        def score(self, item, labels):  # noqa: ANN001, ANN202 - proves the batch path is used.
            raise AssertionError(f"unexpected single-item BioCLIP score for {item.get('detection_id')}")

        def score_label_sets_batch(self, items, label_sets):  # noqa: ANN001, ANN202 - mirrors object batch scorer API.
            if "species" in label_sets:
                raise RuntimeError("invalid BioCLIP tensor shape")
            return {name: [{label: 0.1 for label in labels} for _item in items] for name, labels in label_sets.items()}

    context = _context()
    candidate_set = build_candidate_set(context, allow_single_target_fixture=True)
    work_items = []
    for index in range(2):
        payload = bioclip_score_work_item(
            _detection_row(f"photo-{index}", f"det-{index}", f"sha256:crop-{index}", "butterfly_like", "detected"),
            run_id="run-1",
            detection_shard_uri="s3://biominer/detections.parquet",
            model={"model_id": "fake-bioclip", "model_version": "test", "checkpoint": "fake-checkpoint"},
            candidate_set_id=candidate_set.candidate_set_id,
            ablation_mode="detector_crop",
        )
        work_items.append({"work_key": payload["work_key"], "payload": payload})

    with pytest.raises(RuntimeError, match="invalid BioCLIP tensor shape"):
        run_cloud_bioclip_batch(
            work_items=work_items,
            species_context=context,
            candidate_set=candidate_set,
            scorer=NonMemoryScorer(),
            crop_batch_size=2,
            adaptive_batching=True,
        )


def test_run_cloud_bioclip_batch_adaptive_batching_reports_min_batch_failure() -> None:
    class AlwaysMemoryScorer:
        model_id = "fake-bioclip"
        model_version = "test"
        model_checkpoint = "fake-checkpoint"

        def score(self, item, labels):  # noqa: ANN001, ANN202 - proves the batch path is used.
            raise AssertionError(f"unexpected single-item BioCLIP score for {item.get('detection_id')}")

        def score_label_sets_batch(self, items, label_sets):  # noqa: ANN001, ANN202 - mirrors object batch scorer API.
            if "species" in label_sets:
                raise RuntimeError(f"CUDA out of memory at batch size {len(items)}")
            return {name: [{label: 0.1 for label in labels} for _item in items] for name, labels in label_sets.items()}

    context = _context()
    candidate_set = build_candidate_set(context, allow_single_target_fixture=True)
    work_items = []
    for index in range(2):
        payload = bioclip_score_work_item(
            _detection_row(f"photo-{index}", f"det-{index}", f"sha256:crop-{index}", "butterfly_like", "detected"),
            run_id="run-1",
            detection_shard_uri="s3://biominer/detections.parquet",
            model={"model_id": "fake-bioclip", "model_version": "test", "checkpoint": "fake-checkpoint"},
            candidate_set_id=candidate_set.candidate_set_id,
            ablation_mode="detector_crop",
        )
        work_items.append({"work_key": payload["work_key"], "payload": payload})

    with pytest.raises(RuntimeError, match="CUDA out of memory at batch size 1"):
        run_cloud_bioclip_batch(
            work_items=work_items,
            species_context=context,
            candidate_set=candidate_set,
            scorer=AlwaysMemoryScorer(),
            crop_batch_size=2,
            adaptive_batching=True,
            min_crop_batch_size=1,
        )


def test_run_cloud_bioclip_batch_hierarchical_mode_requires_taxonomy_store() -> None:
    class FailingScorer:
        model_id = "fake-bioclip"
        model_version = "test"
        model_checkpoint = "fake-checkpoint"

        def score(self, item, labels):  # noqa: ANN001, ANN202 - should not be reached.
            raise AssertionError("hierarchical mode must not run target-scope object scoring")

    context = _context()
    candidate_set = build_candidate_set(context, allow_single_target_fixture=True)
    payload = bioclip_score_work_item(
        _detection_row("photo-1", "det-1", "sha256:crop-1", "butterfly_like", "detected"),
        run_id="run-1",
        detection_shard_uri="s3://biominer/detections.parquet",
        model={"model_id": "fake-bioclip", "model_version": "test", "checkpoint": "fake-checkpoint"},
        candidate_set_id=candidate_set.candidate_set_id,
        ablation_mode="detector_crop",
        classification_mode=HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        taxonomy_table_version=CLASSIFICATION_TABLE_VERSION,
        taxonomy_prompt_variant_version=PROMPT_VARIANT_VERSION,
    )

    try:
        run_cloud_bioclip_batch(
            work_items=[{"work_key": payload["work_key"], "payload": payload}],
            species_context=context,
            candidate_set=candidate_set,
            scorer=FailingScorer(),
            classification_mode="hierarchical_butterfly_classification",
        )
    except ValueError as exc:
        assert "taxonomy_store is required" in str(exc)
    else:  # pragma: no cover - assertion branch.
        raise AssertionError("expected hierarchical mode to require taxonomy_store")


def test_run_cloud_bioclip_batch_v3_uses_cached_path_cascade_once_per_crop_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EmbeddingScorer:
        model_id = "fake-bioclip"
        model_version = "test"
        model_checkpoint = "fake-checkpoint"

        def __init__(self) -> None:
            self.image_batches: list[tuple[str, ...]] = []

        def embed_image_items(self, items):  # noqa: ANN001, ANN201 - cascade protocol fake.
            self.image_batches.append(tuple(str(item["detection_id"]) for item in items))
            return [[1.0, 0.0] for _item in items]

    path_store = SimpleNamespace(
        classification_version=CLASSIFICATION_V3_VERSION,
        prompt_version=CLASSIFICATION_V3_PROMPT_VERSION,
    )
    embedding_index = object()
    cascade_calls: list[tuple[str, ...]] = []

    def fake_classify_path_cascade_batch(
        *,
        items,
        embedding_scorer,
        taxonomy_store,
        taxonomy_text_embedding_index,
        rank_beam_width,
    ):  # noqa: ANN001, ANN201 - dependency-boundary fake.
        assert taxonomy_store is path_store
        assert taxonomy_text_embedding_index is embedding_index
        assert rank_beam_width == 3
        cascade_calls.append(tuple(str(item["detection_id"]) for item in items))
        embeddings = embedding_scorer.embed_image_items(items)
        return tuple({"embedding": embedding} for embedding in embeddings)

    def fake_result_to_row(*, item, result, scorer):  # noqa: ANN001, ANN201 - serializer fake.
        assert result == {"embedding": [1.0, 0.0]}
        assert scorer.model_checkpoint == "fake-checkpoint"
        return {
            "source": item["source"],
            "flickr_photo_id": item["flickr_photo_id"],
            "detection_id": item["detection_id"],
            "crop_hash": item["crop_hash"],
            "classification_mode": HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
            "taxonomy_table_version": CLASSIFICATION_V3_VERSION,
            "taxonomy_prompt_variant_version": CLASSIFICATION_V3_PROMPT_VERSION,
        }

    monkeypatch.setattr(
        cloud_work,
        "classify_path_cascade_batch",
        fake_classify_path_cascade_batch,
    )
    monkeypatch.setattr(
        cloud_work,
        "path_cascade_result_to_object_score_row",
        fake_result_to_row,
    )
    monkeypatch.setattr(
        cloud_work,
        "classify_five_rank_crops_batch",
        lambda **_kwargs: pytest.fail("classification-v3 must not use five-rank dispatch"),
    )
    context = _context()
    candidate_set = build_candidate_set(context, allow_single_target_fixture=True)
    work_items = [
        _hierarchical_work_item(
            index,
            candidate_set_id=candidate_set.candidate_set_id,
            taxonomy_table_version=CLASSIFICATION_V3_VERSION,
            taxonomy_prompt_variant_version=CLASSIFICATION_V3_PROMPT_VERSION,
        )
        for index in range(5)
    ]
    scorer = EmbeddingScorer()

    result = run_cloud_bioclip_batch(
        work_items=work_items,
        species_context=context,
        candidate_set=candidate_set,
        scorer=scorer,
        crop_batch_size=2,
        classification_mode=HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        path_taxonomy_store=path_store,  # type: ignore[arg-type]
        taxonomy_text_embedding_index=embedding_index,  # type: ignore[arg-type]
    )

    assert cascade_calls == [("det-0", "det-1"), ("det-2", "det-3"), ("det-4",)]
    assert scorer.image_batches == cascade_calls
    assert result.crops_scored == 5
    assert result.frame["taxonomy_table_version"].unique().to_list() == [
        CLASSIFICATION_V3_VERSION
    ]


def test_run_cloud_bioclip_batch_v3_requires_embedding_index() -> None:
    context = _context()
    candidate_set = build_candidate_set(context, allow_single_target_fixture=True)
    work_item = _hierarchical_work_item(
        1,
        candidate_set_id=candidate_set.candidate_set_id,
        taxonomy_table_version=CLASSIFICATION_V3_VERSION,
        taxonomy_prompt_variant_version=CLASSIFICATION_V3_PROMPT_VERSION,
    )
    path_store = SimpleNamespace(
        classification_version=CLASSIFICATION_V3_VERSION,
        prompt_version=CLASSIFICATION_V3_PROMPT_VERSION,
    )

    with pytest.raises(ValueError, match="taxonomy_text_embedding_index is required"):
        run_cloud_bioclip_batch(
            work_items=[work_item],
            species_context=context,
            candidate_set=candidate_set,
            scorer=_StaticBatchScorer({}),
            classification_mode=HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
            path_taxonomy_store=path_store,  # type: ignore[arg-type]
        )


def test_run_cloud_bioclip_batch_v3_requires_path_store_for_embedding_index() -> None:
    context = _context()
    candidate_set = build_candidate_set(context, allow_single_target_fixture=True)
    work_item = _hierarchical_work_item(
        1,
        candidate_set_id=candidate_set.candidate_set_id,
        taxonomy_table_version=CLASSIFICATION_V3_VERSION,
        taxonomy_prompt_variant_version=CLASSIFICATION_V3_PROMPT_VERSION,
    )

    with pytest.raises(ValueError, match="path_taxonomy_store is required"):
        run_cloud_bioclip_batch(
            work_items=[work_item],
            species_context=context,
            candidate_set=candidate_set,
            scorer=_StaticBatchScorer({}),
            classification_mode=HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
            taxonomy_text_embedding_index=object(),  # type: ignore[arg-type]
        )


def test_run_cloud_bioclip_batch_rejects_payload_classification_mode_mismatch() -> None:
    class FailingScorer:
        model_id = "fake-bioclip"
        model_version = "test"
        model_checkpoint = "fake-checkpoint"

        def score(self, item, labels):  # noqa: ANN001, ANN202 - should not be reached.
            raise AssertionError("cloud worker must reject stale mode before scoring")

    context = _context()
    candidate_set = build_candidate_set(context, allow_single_target_fixture=True)
    payload = bioclip_score_work_item(
        _detection_row("photo-1", "det-1", "sha256:crop-1", "butterfly_like", "detected"),
        run_id="run-1",
        detection_shard_uri="s3://biominer/detections.parquet",
        model={"model_id": "fake-bioclip", "model_version": "test", "checkpoint": "fake-checkpoint"},
        candidate_set_id=candidate_set.candidate_set_id,
        ablation_mode="detector_crop",
        classification_mode=HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        taxonomy_table_version=CLASSIFICATION_TABLE_VERSION,
        taxonomy_prompt_variant_version=PROMPT_VARIANT_VERSION,
    )

    with pytest.raises(ValueError, match="classification_mode"):
        run_cloud_bioclip_batch(
            work_items=[{"work_key": payload["work_key"], "payload": payload}],
            species_context=context,
            candidate_set=candidate_set,
            scorer=FailingScorer(),
        )


def test_run_cloud_bioclip_batch_rejects_payload_top_k_mismatch() -> None:
    store = _butterfly_taxonomy_store()
    context = _context()
    candidate_set = build_candidate_set(context, allow_single_target_fixture=True)
    payload = bioclip_score_work_item(
        _detection_row("photo-1", "det-1", "sha256:crop-1", "butterfly_like", "detected"),
        run_id="run-1",
        detection_shard_uri="s3://biominer/detections.parquet",
        model={"model_id": "fake-bioclip", "model_version": "test", "checkpoint": "fake-checkpoint"},
        candidate_set_id=candidate_set.candidate_set_id,
        ablation_mode="detector_crop",
        classification_mode=HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        taxonomy_table_version=CLASSIFICATION_TABLE_VERSION,
        taxonomy_prompt_variant_version=PROMPT_VARIANT_VERSION,
        species_rerank_top_k=5,
    )

    with pytest.raises(ValueError, match="species_rerank_top_k"):
        run_cloud_bioclip_batch(
            work_items=[{"work_key": payload["work_key"], "payload": payload}],
            species_context=context,
            candidate_set=candidate_set,
            scorer=_StaticBatchScorer({}),
            classification_mode=HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
            taxonomy_store=store,
            species_rerank_top_k=1,
        )


def test_run_cloud_bioclip_batch_rejects_payload_taxonomy_version_mismatch() -> None:
    store = _butterfly_taxonomy_store()
    context = _context()
    candidate_set = build_candidate_set(context, allow_single_target_fixture=True)
    payload = bioclip_score_work_item(
        _detection_row("photo-1", "det-1", "sha256:crop-1", "butterfly_like", "detected"),
        run_id="run-1",
        detection_shard_uri="s3://biominer/detections.parquet",
        model={"model_id": "fake-bioclip", "model_version": "test", "checkpoint": "fake-checkpoint"},
        candidate_set_id=candidate_set.candidate_set_id,
        ablation_mode="detector_crop",
        classification_mode=HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        taxonomy_table_version="classification-table-v-old",
        taxonomy_prompt_variant_version=PROMPT_VARIANT_VERSION,
    )

    with pytest.raises(ValueError, match="taxonomy_table_version"):
        run_cloud_bioclip_batch(
            work_items=[{"work_key": payload["work_key"], "payload": payload}],
            species_context=context,
            candidate_set=candidate_set,
            scorer=_StaticBatchScorer({}),
            classification_mode=HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
            taxonomy_store=store,
        )




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


def _hierarchical_work_item(
    index: int,
    *,
    candidate_set_id: str,
    taxonomy_table_version: str,
    taxonomy_prompt_variant_version: str,
) -> dict[str, object]:
    payload = bioclip_score_work_item(
        _detection_row(
            f"photo-{index}",
            f"det-{index}",
            f"sha256:crop-{index}",
            "butterfly_like",
            "detected",
        ),
        run_id="run-1",
        detection_shard_uri="s3://biominer/detections.parquet",
        model={
            "model_id": "fake-bioclip",
            "model_version": "test",
            "checkpoint": "fake-checkpoint",
        },
        candidate_set_id=candidate_set_id,
        ablation_mode="detector_crop",
        classification_mode=HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        taxonomy_table_version=taxonomy_table_version,
        taxonomy_prompt_variant_version=taxonomy_prompt_variant_version,
    )
    return {"work_key": payload["work_key"], "payload": payload}


class _StaticBatchScorer:
    model_id = "fake-bioclip"
    model_version = "test"
    model_checkpoint = "fake-checkpoint"

    def __init__(self, scores_by_crop: dict[str, dict[str, float]]) -> None:
        self._scores_by_crop = scores_by_crop

    def score(self, item: dict[str, object], labels: tuple[str, ...]) -> dict[str, float]:
        scores = self._scores_by_crop.get(str(item.get("crop_hash") or ""), {})
        return {label: float(scores.get(label, 0.0)) for label in labels}

    def score_label_sets_batch(
        self,
        items: list[dict[str, object]],
        label_sets: dict[str, tuple[str, ...]],
    ) -> dict[str, list[dict[str, float]]]:
        return {
            name: [self.score(item, tuple(labels)) for item in items]
            for name, labels in label_sets.items()
        }


def _butterfly_taxonomy_store():
    return benchmark_taxonomy_store()


class _FakeCloudStorage:
    def __init__(self) -> None:
        self.parquet_payloads: dict[str, pl.DataFrame] = {}

    def read_parquet(self, uri: str) -> pl.DataFrame:
        return self.parquet_payloads[uri]

    def iter_parquet_batches(self, uri: str, *, batch_size: int):  # noqa: ANN201 - fake protocol implementation.
        yield from self.parquet_payloads[uri].iter_slices(batch_size)
