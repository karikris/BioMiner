"""Tests for the canonical detector and routing entry contract."""

from __future__ import annotations

from dataclasses import replace
import json

import pytest

from biominer.detection.cloud_work import (
    detection_work_item,
    run_cloud_detection_batch,
)
from biominer.detection.detector_base import DecodedImage, FakeObjectDetector
from biominer.detection.pipeline import run_detection_pipeline
from biominer.detection.policy import DetectionPolicy
from biominer.detection.route_contract import (
    DetectorRouteContract,
    build_detector_route_contract,
)
from biominer.detection.yoloe26_detector import (
    default_yoloe26_prompts,
    yoloe26_prompt_set_fingerprint,
)


def test_injected_detector_contract_round_trips_and_binds_routing_policy() -> None:
    detector = FakeObjectDetector()
    policy = DetectionPolicy(backend="fake")

    contract = build_detector_route_contract(detector, policy)
    restored = DetectorRouteContract.from_mapping(
        json.loads(json.dumps(contract.to_dict(), sort_keys=True))
    )

    assert restored == contract
    assert restored.execution_mode == "injected"
    assert restored.routing_policy_fingerprint == policy.routing_policy.fingerprint
    assert restored.fingerprint == contract.fingerprint
    changed = build_detector_route_contract(
        detector, replace(policy, box_score_threshold=0.21)
    )
    assert changed.fingerprint != contract.fingerprint


def test_yoloe_contract_requires_exact_model_prompt_and_threshold_identity() -> None:
    prompts = default_yoloe26_prompts()
    detector = {
        "backend": "yoloe26",
        "model_id": "yoloe26:yoloe-26s-seg",
        "model_version": "ultralytics:test",
        "checkpoint": "yoloe-26s-seg.pt",
        "prompt_classes": list(prompts),
        "prompt_set_fingerprint": yoloe26_prompt_set_fingerprint(prompts),
        "execution_mode": "persistent_sidecar",
        "transport": "json_b64",
        "imgsz": 768,
        "conf": 0.20,
        "iou": 0.50,
        "max_det": 8,
    }
    policy = DetectionPolicy(backend="yoloe26")

    contract = build_detector_route_contract(detector, policy)

    assert contract.backend == "yoloe26"
    assert contract.execution_mode == "persistent_sidecar"
    assert contract.detector_image_size == 768
    assert contract.prompt_set_fingerprint == yoloe26_prompt_set_fingerprint(
        prompts
    )
    with pytest.raises(ValueError, match="confidence thresholds differ"):
        build_detector_route_contract({**detector, "conf": 0.25}, policy)
    with pytest.raises(ValueError, match="fingerprint does not match"):
        build_detector_route_contract(
            {**detector, "prompt_set_fingerprint": "sha256:" + "0" * 64},
            policy,
        )


def test_local_pipeline_rejects_backend_drift_and_reports_contract(tmp_path) -> None:
    detector = FakeObjectDetector()
    record = {
        "source": "flickr",
        "flickr_photo_id": "photo-1",
        "source_record_hash": "sha256:source-1",
        "image_url": "memory://photo-1",
    }

    with pytest.raises(ValueError, match="backend.*differ"):
        run_detection_pipeline(
            records=[record],
            detector=detector,
            output_path=tmp_path / "mismatch.parquet",
            image_loader=lambda _record: _image(),
            detection_policy=DetectionPolicy(backend="yoloe26"),
        )

    result = run_detection_pipeline(
        records=[record],
        detector=detector,
        output_path=tmp_path / "detections.parquet",
        image_loader=lambda _record: _image(),
        detection_policy=DetectionPolicy(backend="fake"),
    )

    expected = build_detector_route_contract(
        detector, DetectionPolicy(backend="fake")
    )
    assert result.detector_route_contract_fingerprint == expected.fingerprint
    assert result.detector_route_contract_version == expected.contract_version
    assert result.detector_execution_mode == "injected"
    assert result.detector_model_load_count == 0


def test_cloud_batch_fails_closed_on_queued_detector_or_policy_drift() -> None:
    payload = detection_work_item(
        {
            "source": "flickr",
            "flickr_photo_id": "photo-1",
            "source_record_hash": "sha256:source-1",
            "image_url": "memory://photo-1",
        },
        run_id="run-1",
        source_shard_uri="s3://bucket/source.parquet",
        detector={
            "backend": "fake",
            "model_id": "fake-detector",
            "model_version": "test",
            "checkpoint": "fake-checkpoint",
        },
        detection_policy=DetectionPolicy(backend="fake"),
    )
    work_item = {"work_key": payload["work_key"], "payload": payload}

    assert payload["detector_route_contract"]["contract_fingerprint"]

    class DriftedDetector(FakeObjectDetector):
        checkpoint = "different-checkpoint"

    with pytest.raises(ValueError, match="differs from queued work"):
        run_cloud_detection_batch(
            work_items=[work_item],
            detector=DriftedDetector(),
            image_loader=lambda _record: _image(),
            detection_policy=DetectionPolicy(backend="fake"),
        )

    with pytest.raises(ValueError, match="differs from queued work"):
        run_cloud_detection_batch(
            work_items=[work_item],
            detector=FakeObjectDetector(),
            image_loader=lambda _record: _image(),
            detection_policy=DetectionPolicy(
                backend="fake", box_score_threshold=0.25
            ),
        )


def _image() -> DecodedImage:
    return DecodedImage(
        width=2,
        height=2,
        mode="RGB",
        data=bytes([255, 0, 0] * 4),
        source_uri="memory://image",
    )
