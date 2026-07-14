from __future__ import annotations

from pathlib import Path

import polars as pl

from biominer.detection.detector_base import DecodedImage
from biominer.vision.gates import BioClipGateMode, BioClipGatePolicy
from biominer.vision.score_inputs import BIOCLIP_SCORE_INPUT_SCHEMA, materialize_bioclip_score_inputs
from factories import canonical_records, flickr_source_record, object_detection_row


_ROUTING_POLICY_FINGERPRINT = "sha256:" + "b" * 64


def test_materialize_bioclip_score_inputs_scores_non_hard_and_no_detection_fallback(tmp_path: Path) -> None:
    records = canonical_records(
        flickr_source_record("photo-butterfly"),
        flickr_source_record("photo-moth"),
        flickr_source_record("photo-hard-negative"),
        flickr_source_record("photo-empty"),
        flickr_source_record("photo-failed"),
    )
    detections = pl.DataFrame(
        [
            object_detection_row("photo-butterfly", detection_id="det-butterfly", label="butterfly_like", crop_hash="sha256:butterfly"),
            object_detection_row("photo-moth", detection_id="det-moth", label="moth_like", crop_hash="sha256:moth"),
            object_detection_row("photo-hard-negative", detection_id="det-hard-negative", label="hard_negative", crop_hash="sha256:hard"),
            object_detection_row(
                "photo-empty",
                detection_id="det-empty",
                label="no_detection",
                crop_hash="",
                detection_status="no_detection",
                failure_reason="no_butterfly_like_object",
            ),
            object_detection_row(
                "photo-failed",
                detection_id="det-failed",
                label="failed_image_load",
                crop_hash="",
                detection_status="failed_image_load",
                failure_reason="decode failed",
            ),
        ]
    )
    loaded_for: list[str] = []

    def image_loader(item: dict[str, object]) -> DecodedImage:
        loaded_for.append(str(item["detection_id"]))
        return _decoded_image()

    result = materialize_bioclip_score_inputs(
        canonical_records=records,
        detections=detections,
        image_loader=image_loader,
        temp_dir=tmp_path,
        gate_policy=BioClipGatePolicy(
            mode=BioClipGateMode.EXCLUDE_HARD_NEGATIVE,
            score_no_detection_whole_image=True,
        ),
        crop_target_px=3,
        batch_id="batch-001",
        part_id="part-000001",
    )

    assert result.frame.schema == BIOCLIP_SCORE_INPUT_SCHEMA
    assert result.frame["detection_id"].to_list() == ["det-butterfly", "det-moth", "det-empty"]
    assert result.frame["visual_input_kind"].to_list() == ["detector_crop", "detector_crop", "whole_image"]
    assert result.frame["bioclip_gate_reason"].to_list() == [
        "detected_non_hard_negative",
        "detected_non_hard_negative",
        "no_detection_whole_image_fallback",
    ]
    assert loaded_for == ["det-butterfly", "det-moth", "det-empty"]
    assert all(path.exists() for path in result.paths)
    assert [item["ablation_mode"] for item in result.items] == ["detector_crop", "detector_crop", "whole_image"]

    result.cleanup()

    assert not result.temp_dir.exists()


def test_materialize_routed_score_inputs_propagates_route_and_policy_identity(
    tmp_path: Path,
) -> None:
    records = canonical_records(
        flickr_source_record("photo-adult"),
        flickr_source_record("photo-review"),
    )
    detections = pl.DataFrame(
        [
            {
                **object_detection_row(
                    "photo-adult",
                    detection_id="det-adult",
                    crop_hash="sha256:adult",
                ),
                **_routing_fields(
                    detection_route="adult_butterfly_field",
                    routing_action="score",
                    bioclip_route="adult_field",
                    routing_priority="standard",
                ),
            },
            {
                **object_detection_row(
                    "photo-review",
                    detection_id="det-review",
                    crop_hash="sha256:review",
                ),
                **_routing_fields(
                    detection_route="ambiguous_visual_domain",
                    routing_action="review",
                    bioclip_route="larval",
                    routing_priority="low",
                ),
            },
        ]
    )
    loaded_for: list[str] = []

    def image_loader(item: dict[str, object]) -> DecodedImage:
        loaded_for.append(str(item["detection_id"]))
        return _decoded_image()

    result = materialize_bioclip_score_inputs(
        canonical_records=records,
        detections=detections,
        image_loader=image_loader,
        temp_dir=tmp_path,
        crop_target_px=3,
    )

    assert result.frame.schema == BIOCLIP_SCORE_INPUT_SCHEMA
    assert result.frame["detection_id"].to_list() == ["det-adult"]
    assert result.frame.select(
        "detection_route",
        "routing_action",
        "bioclip_route",
        "routing_priority",
        "routing_reason",
        "routing_policy_version",
        "routing_policy_fingerprint",
    ).row(0, named=True) == _routing_fields(
        detection_route="adult_butterfly_field",
        routing_action="score",
        bioclip_route="adult_field",
        routing_priority="standard",
    )
    assert {
        name: result.items[0][name]
        for name in (
            "detection_route",
            "routing_action",
            "bioclip_route",
            "routing_priority",
            "routing_reason",
            "routing_policy_version",
            "routing_policy_fingerprint",
        )
    } == _routing_fields(
        detection_route="adult_butterfly_field",
        routing_action="score",
        bioclip_route="adult_field",
        routing_priority="standard",
    )
    assert loaded_for == ["det-adult"]

    result.cleanup()


def _routing_fields(
    *,
    detection_route: str,
    routing_action: str,
    bioclip_route: str,
    routing_priority: str,
) -> dict[str, object]:
    return {
        "detection_route": detection_route,
        "routing_action": routing_action,
        "bioclip_route": bioclip_route,
        "routing_priority": routing_priority,
        "routing_reason": "test routing decision",
        "routing_policy_version": "detection-routing-v1",
        "routing_policy_fingerprint": _ROUTING_POLICY_FINGERPRINT,
    }


def _decoded_image() -> DecodedImage:
    pixels = bytes(
        value
        for y in range(4)
        for x in range(4)
        for value in ((x * 40) % 256, (y * 40) % 256, ((x + y) * 20) % 256)
    )
    return DecodedImage(width=4, height=4, mode="RGB", data=pixels, source_uri="memory://score-input")
