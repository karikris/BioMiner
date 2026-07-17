from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

import biominer.bioclip.cascade_contract as cascade_contract
from biominer.bioclip.classification_modes import HIERARCHICAL_BUTTERFLY_CLASSIFICATION
from biominer.registry.classification_v3 import (
    CLASSIFICATION_RANKS,
    CLASSIFICATION_V3_PROMPT_VERSION,
    CLASSIFICATION_V3_VERSION,
)
from biominer.detection.policy import VisionRuntimeSettings
from biominer.storage.parquet import ParquetPartWrite
from biominer.vision.cloud_work import (
    ROLLING_VISION_ARTIFACT_ORDER,
    ROLLING_VISION_ARTIFACT_STAGES,
    commit_rolling_vision_batch_shards,
    enqueue_rolling_vision_work_from_source_shards,
    rolling_vision_settings_key,
    rolling_vision_work_item,
)
from biominer.workstore.sqlite import SQLiteWorkStore


@pytest.mark.parametrize(
    "identity_field",
    (
        "contract_version",
        "beam_strategy",
        "rank_beam_width",
        "rank_order",
        "classification_version",
        "prompt_version",
        "taxonomy_fingerprint",
        "hierarchy_fingerprint",
        "embedding_cache_fingerprint",
        "species_first_pass_top_k",
        "species_rerank_top_k",
        "species_report_top_k",
        "species_rerank_prompt_version",
    ),
)
def test_rolling_vision_work_key_changes_by_every_v3_cascade_identity_field(
    identity_field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        {
            "source": "flickr",
            "flickr_photo_id": "photo-1",
            "source_record_hash": "sha256:source-1",
            "image_url": "https://live.staticflickr.com/photo-1.jpg",
        }
    ]
    detector = {
        "backend": "yoloe26",
        "model_id": "yoloe26",
        "model_version": "test",
        "checkpoint": "yoloe-26s-seg.pt",
    }
    bioclip_model = {
        "model_id": "bioclip-2.5",
        "model_version": "test",
        "checkpoint": "hf-hub:imageomics/bioclip-2.5-vith14",
    }
    base_identity = _v3_cascade_identity()
    base_settings = rolling_vision_settings_key(
        detector=detector,
        vision_settings=VisionRuntimeSettings(
            yolo_imgsz=768, yolo_conf=0.20, yolo_iou=0.50, yolo_max_det=8
        ),
        bioclip_model=bioclip_model,
        candidate_set_id="candidate-set-v1",
        classification_mode=HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        cascade_identity=base_identity,
    )
    changed_identity = _changed_v3_cascade_identity(identity_field, monkeypatch)
    changed_settings = rolling_vision_settings_key(
        detector=detector,
        vision_settings=VisionRuntimeSettings(
            yolo_imgsz=768, yolo_conf=0.20, yolo_iou=0.50, yolo_max_det=8
        ),
        bioclip_model=bioclip_model,
        candidate_set_id="candidate-set-v1",
        classification_mode=HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        cascade_identity=changed_identity,
    )

    base = rolling_vision_work_item(
        records,
        run_id="run-1",
        batch_index=0,
        vision_batch_rows=500,
        source_shard_uris=["s3://biominer/source.parquet"],
        settings_key=base_settings,
    )
    changed = rolling_vision_work_item(
        records,
        run_id="run-1",
        batch_index=0,
        vision_batch_rows=500,
        source_shard_uris=["s3://biominer/source.parquet"],
        settings_key=changed_settings,
    )

    assert base["work_key"] != changed["work_key"]
    assert base["settings_key"]["detector"]["yolo_imgsz"] == 768
    assert base["settings_key"]["crop"]["crop_target_px"] == 336
    assert base["settings_key"]["bioclip_gate"]["mode"] == "routed_visual_domain"
    assert (
        base["settings_key"]["bioclip_model"]["checkpoint"]
        == "hf-hub:imageomics/bioclip-2.5-vith14"
    )
    assert (
        base["settings_key"]["classification_mode"]
        == HIERARCHICAL_BUTTERFLY_CLASSIFICATION
    )
    assert base["settings_key"]["cascade_identity"] == base_identity
    assert tuple(base_identity["rank_order"]) == CLASSIFICATION_RANKS
    assert len(base_identity["rank_order"]) == 6
    assert "top_k_settings" not in base["settings_key"]
    assert "family_top_k" not in base_identity


def test_rolling_vision_work_key_ignores_retry_and_attempt_metadata() -> None:
    record = {
        "source": "flickr",
        "flickr_photo_id": "photo-1",
        "source_record_hash": "sha256:source-1",
        "image_url": "https://live.staticflickr.com/photo-1.jpg",
    }
    settings = rolling_vision_settings_key(
        detector={
            "backend": "yoloe26",
            "model_id": "yoloe26",
            "model_version": "test",
            "checkpoint": "ckpt",
        },
        vision_settings=VisionRuntimeSettings(yolo_imgsz=768),
        bioclip_model={
            "model_id": "bioclip",
            "model_version": "test",
            "checkpoint": "model",
        },
        candidate_set_id="candidate-set-v1",
        classification_mode=HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        cascade_identity=_v3_cascade_identity(),
    )
    base = rolling_vision_work_item(
        [record],
        run_id="run-1",
        batch_index=0,
        vision_batch_rows=500,
        source_shard_uris=["s3://biominer/source.parquet"],
        settings_key=settings,
    )
    retried = rolling_vision_work_item(
        [
            {
                **record,
                "attempt_count": 3,
                "retry_count": 2,
                "last_error": "transient timeout",
            }
        ],
        run_id="run-1",
        batch_index=0,
        vision_batch_rows=500,
        source_shard_uris=["s3://biominer/source.parquet"],
        settings_key=settings,
    )

    assert retried["work_key"] == base["work_key"]


def test_rolling_vision_settings_default_to_adult_routed_gate_and_bind_prompt_policy() -> None:
    settings = rolling_vision_settings_key(
        detector={
            "backend": "yoloe26",
            "model_id": "yoloe26",
            "model_version": "test",
            "checkpoint": "yoloe-26s-seg.pt",
            "prompt_classes": ["butterfly", "moth"],
            "prompt_set_fingerprint": "sha256:" + "a" * 64,
        },
        vision_settings=VisionRuntimeSettings(
            possible_adult_route_enabled=True,
            possible_adult_route_threshold=0.25,
            ambiguous_insect_review_enabled=False,
            ambiguous_insect_review_threshold=0.20,
        ),
        bioclip_model={
            "model_id": "bioclip",
            "model_version": "test",
            "checkpoint": "model",
        },
        candidate_set_id="candidate-set-v1",
    )

    assert settings["bioclip_gate"] == {
        "mode": "routed_visual_domain",
        "supported_comparison_routes": ["adult_field"],
    }
    assert settings["detector"]["prompt_classes"] == ["butterfly", "moth"]
    assert settings["detector"]["prompt_set_fingerprint"] == "sha256:" + "a" * 64
    assert settings["detector"]["routing_policy"] == {
        "version": "detection-routing-policy-v1",
        "fingerprint": settings["detector"]["routing_policy"]["fingerprint"],
        "possible_adult_route_enabled": True,
        "possible_adult_route_threshold": 0.25,
        "ambiguous_insect_review_enabled": False,
        "ambiguous_insect_review_threshold": 0.20,
    }
    assert settings["detector"]["routing_policy"]["fingerprint"].startswith(
        "sha256:"
    )


def test_enqueue_rolling_vision_work_batches_source_shards_deterministically(tmp_path: Path) -> None:
    storage = _FakeCloudStorage()
    workstore = SQLiteWorkStore(tmp_path / "workstore.sqlite")
    source_a = "s3://biominer/source/a.parquet"
    source_b = "s3://biominer/source/b.parquet"
    storage.parquet_payloads[source_b] = pl.DataFrame([_source_record("photo-b")])
    storage.parquet_payloads[source_a] = pl.DataFrame([_source_record("photo-a")])
    for uri in (source_b, source_a):
        workstore.register_shard(
            job_name="biominer_production_run",
            registry_version="registry-v1",
            stage="poll_flickr",
            run_id="run-1",
            worker_id="poller",
            uri=uri,
            checksum=None,
            row_count=1,
        )

    first = enqueue_rolling_vision_work_from_source_shards(
        storage=storage,
        workstore=workstore,
        job_name="biominer_production_run",
        registry_version="registry-v1",
        run_id="run-1",
        source_stage="poll_flickr",
        vision_batch_rows=1,
        detector={"backend": "yoloe26", "model_id": "yoloe26", "model_version": "test", "checkpoint": "ckpt"},
        vision_settings=VisionRuntimeSettings(yolo_imgsz=768),
        bioclip_model={"model_id": "bioclip", "model_version": "test", "checkpoint": "model"},
    )
    second = enqueue_rolling_vision_work_from_source_shards(
        storage=storage,
        workstore=workstore,
        job_name="biominer_production_run",
        registry_version="registry-v1",
        run_id="run-1",
        source_stage="poll_flickr",
        vision_batch_rows=1,
        detector={"backend": "yoloe26", "model_id": "yoloe26", "model_version": "test", "checkpoint": "ckpt"},
        vision_settings=VisionRuntimeSettings(yolo_imgsz=768),
        bioclip_model={"model_id": "bioclip", "model_version": "test", "checkpoint": "model"},
    )

    assert first.source_shards_seen == 2
    assert first.source_records_seen == 2
    assert first.batches_planned == 2
    assert first.enqueued_work_items == 2
    assert second.enqueued_work_items == 0
    assert second.duplicate_work_items == 2
    items = workstore.list_work_items(
        job_name="biominer_production_run",
        stage="detect_objects",
        registry_version="registry-v1",
    )
    assert [item["payload"]["batch_id"] for item in items] == ["vision-batch-000000", "vision-batch-000001"]
    assert [item["payload"]["source_records"][0]["flickr_photo_id"] for item in items] == ["photo-a", "photo-b"]


def test_commit_rolling_vision_shards_writes_all_parts_before_registering_and_completing() -> None:
    storage = _RecordingStorage()
    workstore = _RecordingWorkStore()
    frames = {artifact: pl.DataFrame({"artifact": [artifact]}) for artifact in ROLLING_VISION_ARTIFACT_ORDER}

    result = commit_rolling_vision_batch_shards(
        storage=storage,
        workstore=workstore,
        job_name="biominer_production_run",
        registry_version="registry-v1",
        run_id="run-1",
        worker_id="worker-1",
        base_prefix="s3://biominer/runs/run_id=run-1/staging",
        work_key="run-1:rolling-vision:abc",
        batch_id="vision-batch-000000",
        part_id="part-000000",
        frames=frames,
        compression="zstd",
    )

    first_register_index = workstore.events.index("register:image_batch_manifest")
    assert storage.events == [f"write:{artifact}" for artifact in ROLLING_VISION_ARTIFACT_ORDER]
    assert first_register_index == 0
    assert workstore.events[-1] == "complete:run-1:rolling-vision:abc"
    assert [event.split(":", 1)[1] for event in workstore.events[:-1]] == list(ROLLING_VISION_ARTIFACT_ORDER)
    assert result.parts_written == len(ROLLING_VISION_ARTIFACT_ORDER)
    assert result.parts_reused == 0
    assert result.checkpointed_shards == len(ROLLING_VISION_ARTIFACT_ORDER)
    assert set(result.output_uris) == set(ROLLING_VISION_ARTIFACT_ORDER)
    assert workstore.completed[0]["output_uri"] == result.output_uris["photo_evidence_summary"]
    assert {shard["stage"] for shard in workstore.shards} == set(ROLLING_VISION_ARTIFACT_STAGES.values())


def test_commit_rolling_vision_shards_does_not_register_or_complete_after_write_failure() -> None:
    storage = _RecordingStorage(fail_on_artifact="object_bioclip_scores")
    workstore = _RecordingWorkStore()
    frames = {artifact: pl.DataFrame({"artifact": [artifact]}) for artifact in ROLLING_VISION_ARTIFACT_ORDER}

    with pytest.raises(RuntimeError, match="write failed"):
        commit_rolling_vision_batch_shards(
            storage=storage,
            workstore=workstore,
            job_name="biominer_production_run",
            registry_version="registry-v1",
            run_id="run-1",
            worker_id="worker-1",
            base_prefix="s3://biominer/runs/run_id=run-1/staging",
            work_key="run-1:rolling-vision:abc",
            batch_id="vision-batch-000000",
            part_id="part-000000",
            frames=frames,
        )

    assert storage.events == [
        "write:image_batch_manifest",
        "write:object_detections",
        "write:bioclip_score_inputs",
        "write:object_bioclip_scores",
    ]
    assert workstore.events == []
    assert workstore.shards == []
    assert workstore.completed == []


def _source_record(photo_id: str) -> dict[str, str]:
    return {
        "source": "flickr",
        "flickr_photo_id": photo_id,
        "source_record_hash": f"sha256:{photo_id}",
        "image_url": f"https://live.staticflickr.com/{photo_id}.jpg",
    }


def _v3_cascade_identity(
    *,
    classification_version: str = CLASSIFICATION_V3_VERSION,
    prompt_version: str = CLASSIFICATION_V3_PROMPT_VERSION,
    taxonomy_fingerprint: str = "sha256:classification-v1",
    hierarchy_fingerprint: str = "sha256:hierarchy-v1",
    embedding_cache_fingerprint: str = "sha256:embedding-cache-v1",
) -> dict[str, Any]:
    return cascade_contract.production_cascade_work_identity(
        classification_version=classification_version,
        prompt_version=prompt_version,
        taxonomy_fingerprint=taxonomy_fingerprint,
        hierarchy_fingerprint=hierarchy_fingerprint,
        embedding_cache_fingerprint=embedding_cache_fingerprint,
    )


def _changed_v3_cascade_identity(
    field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    if field == "contract_version":
        monkeypatch.setattr(
            cascade_contract,
            "CASCADE_WORK_IDENTITY_VERSION",
            "butterfly-cascade-work-identity-v-next",
        )
    elif field == "beam_strategy":
        monkeypatch.setattr(
            cascade_contract, "GLOBAL_RANK_TOP_K_BEAM_STRATEGY", "global_rank_top_k_v2"
        )
    elif field == "rank_beam_width":
        monkeypatch.setattr(cascade_contract, "DEFAULT_RANK_BEAM_WIDTH", 4)
    elif field == "rank_order":
        monkeypatch.setattr(
            cascade_contract,
            "CASCADE_RANK_ORDER",
            ("FAMILY", "SUBFAMILY", "TRIBE", "GENUS", "SUBTRIBE", "SPECIES"),
        )
    elif field == "classification_version":
        monkeypatch.setattr(
            cascade_contract,
            "CLASSIFICATION_V3_VERSION",
            "butterfly-classification-v3.1.0",
        )
        return _v3_cascade_identity(
            classification_version="butterfly-classification-v3.1.0"
        )
    elif field == "prompt_version":
        return _v3_cascade_identity(prompt_version="butterfly-six-rank-prompts-v-next")
    elif field == "taxonomy_fingerprint":
        return _v3_cascade_identity(taxonomy_fingerprint="sha256:classification-v2")
    elif field == "hierarchy_fingerprint":
        return _v3_cascade_identity(hierarchy_fingerprint="sha256:hierarchy-v2")
    elif field == "embedding_cache_fingerprint":
        return _v3_cascade_identity(
            embedding_cache_fingerprint="sha256:embedding-cache-v2"
        )
    elif field == "species_first_pass_top_k":
        monkeypatch.setattr(cascade_contract, "DEFAULT_SPECIES_FIRST_PASS_TOP_K", 21)
    elif field == "species_rerank_top_k":
        monkeypatch.setattr(cascade_contract, "DEFAULT_SPECIES_RERANK_TOP_K", 6)
    elif field == "species_report_top_k":
        monkeypatch.setattr(cascade_contract, "DEFAULT_SPECIES_REPORT_TOP_K", 4)
    elif field == "species_rerank_prompt_version":
        monkeypatch.setattr(
            cascade_contract, "SPECIES_RERANK_PROMPT_STAGE", "species_rerank_v2"
        )
    else:  # pragma: no cover - parameter list is exhaustive.
        raise AssertionError(f"unsupported cascade identity field: {field}")
    return _v3_cascade_identity()


class _FakeCloudStorage:
    def __init__(self) -> None:
        self.parquet_payloads: dict[str, pl.DataFrame] = {}

    def read_parquet(self, uri: str) -> pl.DataFrame:
        return self.parquet_payloads[uri]

    def iter_parquet_batches(self, uri: str, *, batch_size: int):  # noqa: ANN201
        yield from self.parquet_payloads[uri].iter_slices(batch_size)


class _RecordingStorage:
    def __init__(self, *, fail_on_artifact: str | None = None) -> None:
        self.fail_on_artifact = fail_on_artifact
        self.events: list[str] = []
        self.parquet_payloads: dict[str, pl.DataFrame] = {}

    def write_parquet_part(
        self,
        uri: str,
        frame: pl.DataFrame,
        *,
        compression: str | None = "zstd",
        overwrite: bool = False,
    ) -> ParquetPartWrite:
        artifact = _artifact_from_uri(uri)
        self.events.append(f"write:{artifact}")
        if artifact == self.fail_on_artifact:
            raise RuntimeError(f"write failed for {artifact}")
        if not overwrite and uri in self.parquet_payloads:
            raise FileExistsError(uri)
        self.parquet_payloads[uri] = frame
        return ParquetPartWrite(uri=uri, row_count=frame.height, byte_count=None, compression=compression)

    def exists(self, uri: str) -> bool:
        return uri in self.parquet_payloads


class _RecordingWorkStore:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.shards: list[dict[str, Any]] = []
        self.completed: list[dict[str, Any]] = []

    def register_shard(self, **kwargs: Any) -> None:
        artifact = str((kwargs.get("metadata") or {}).get("artifact") or "")
        self.events.append(f"register:{artifact}")
        self.shards.append(dict(kwargs))

    def mark_completed(self, work_key: str, output_uri: str | None, checksum: str | None, row_count: int | None) -> None:
        self.events.append(f"complete:{work_key}")
        self.completed.append(
            {
                "work_key": work_key,
                "output_uri": output_uri,
                "checksum": checksum,
                "row_count": row_count,
            }
        )


def _artifact_from_uri(uri: str) -> str:
    marker = "/stage="
    stage = uri.split(marker, 1)[1].split("/", 1)[0]
    return {
        "image_batch_manifest": "image_batch_manifest",
        "object_detections": "object_detections",
        "bioclip_score_inputs": "bioclip_score_inputs",
        "object_bioclip_scores": "object_bioclip_scores",
        "object_evidence_joined": "object_evidence_joined",
        "photo_evidence_summary": "photo_evidence_summary",
    }[stage]
